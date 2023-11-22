from DTOs.person_dto import PersonDTO
from models.person import Person


class PersonMapper:
    @staticmethod
    def to_dto(person: Person) -> PersonDTO:
        return PersonDTO(person.id, person.cropped_face, person.cropped_face)